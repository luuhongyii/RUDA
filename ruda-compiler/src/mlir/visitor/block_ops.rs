use melior::{Context, Error, dialect::arith,
    ir::{BlockLike, Location, Operation, Type, Value, attribute::IntegerAttribute, operation::OperationLike}};

pub trait BlockOps<'c: 'a, 'a>: BlockLike<'c, 'a> {
    fn arg(&self, index: usize) -> Result<Value<'c, 'a>, Error> {
        Ok(self.argument(index)?.into())
    }

    fn append_op_result(&self, operation: impl Into<Operation<'c>>) -> Result<Value<'c, 'a>, Error> {
        Ok(self.append_operation(operation.into()).result(0)?.into())
    }

    fn const_int_from_type(&self, context: &'c Context, location: Location<'c>,
        value: i64, ty: Type<'c>) -> Result<Value<'c, 'a>, Error> {
        self.append_op_result(arith::constant(context, IntegerAttribute::new(ty, value).into(), location))
    }

    fn addi(&self, lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Result<Value<'c, 'a>, Error> {
        self.append_op_result(arith::addi(lhs, rhs, location))
    }

    fn muli(&self, lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Result<Value<'c, 'a>, Error> {
        self.append_op_result(arith::muli(lhs, rhs, location))
    }

    fn extui(&self, value: Value<'c, '_>, ty: Type<'c>, location: Location<'c>) -> Result<Value<'c, 'a>, Error> {
        self.append_op_result(arith::extui(value, ty, location))
    }
}

impl<'c: 'a, 'a, B: BlockLike<'c, 'a>> BlockOps<'c, 'a> for B {}
